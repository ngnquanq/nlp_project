import json
from pathlib import Path

import yaml

from mt_pipeline.reruns import OUTPUT_DIR_KEYS, derive_run_config


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_derive_relocates_only_output_dirs(tmp_path):
    derived = derive_run_config("configs/e1_fairseq.yaml", tmp_path)
    original = _load("configs/e1_fairseq.yaml")
    result = _load(derived)

    for key in OUTPUT_DIR_KEYS:
        assert result[key].startswith(str(tmp_path))
        assert result[key] != original[key]

    # Everything that defines the experiment must survive untouched.
    for key in ("experiment_id", "backend", "seed", "model", "training", "decoding"):
        assert result[key] == original[key]


def test_derive_preserves_relative_structure(tmp_path):
    result = _load(derive_run_config("configs/e1_fairseq.yaml", tmp_path))
    assert result["work_dir"] == f"{tmp_path}/work/e1_fairseq_vi_zh_v1"
    assert result["checkpoint_dir"] == f"{tmp_path}/checkpoint/e1_fairseq_vi_zh_v1"
    assert result["prediction_dir"] == f"{tmp_path}/predictions"


def test_derive_does_not_collapse_work_and_checkpoint_dirs(tmp_path):
    # work/<id> and checkpoint/<id> share a basename; relocating on basename alone
    # merged the training log and the checkpoints into one directory.
    for config in ("configs/e1_fairseq.yaml", "configs/smoke_fairseq.yaml"):
        result = _load(derive_run_config(config, tmp_path / Path(config).stem))
        assert result["work_dir"] != result["checkpoint_dir"]


def test_derive_gpu_flag_forces_cuda_fp16(tmp_path):
    # The smoke config is cpu/fp32; a repro run must take the E1/E3 runtime path.
    baseline = _load("configs/smoke_fairseq.yaml")
    assert baseline["training"]["cpu"] is True
    assert baseline["training"]["fp16"] is False

    result = _load(derive_run_config("configs/smoke_fairseq.yaml", tmp_path, gpu=True))
    assert result["training"]["cpu"] is False
    assert result["training"]["fp16"] is True


def test_derive_without_gpu_flag_leaves_runtime_alone(tmp_path):
    result = _load(derive_run_config("configs/smoke_fairseq.yaml", tmp_path))
    assert result["training"]["cpu"] is True
    assert result["training"]["fp16"] is False


def test_derive_drops_internal_config_path_key(tmp_path):
    result = _load(derive_run_config("configs/e1_fairseq.yaml", tmp_path))
    assert "_config_path" not in result


def test_prediction_comparison_ignores_run_scoped_checkpoint_path(tmp_path):
    from mt_pipeline.reruns import compare_predictions_content

    row = {"sample_id": "s1", "prediction": "字", "checkpoint": "/runA/ckpt.pt"}
    left = tmp_path / "a.jsonl"
    right = tmp_path / "b.jsonl"
    left.write_text(json.dumps(row) + "\n", encoding="utf-8")
    right.write_text(
        json.dumps({**row, "checkpoint": "/runB/ckpt.pt"}) + "\n", encoding="utf-8"
    )

    result = compare_predictions_content(left, right)
    assert result["identical"] is True
    assert result["fields_ignored"] == ["checkpoint"]


def test_prediction_comparison_catches_translation_drift(tmp_path):
    from mt_pipeline.reruns import compare_predictions_content

    row = {"sample_id": "s1", "prediction": "字", "checkpoint": "/runA/ckpt.pt"}
    left = tmp_path / "a.jsonl"
    right = tmp_path / "b.jsonl"
    left.write_text(json.dumps(row) + "\n", encoding="utf-8")
    right.write_text(
        json.dumps({**row, "prediction": "他"}) + "\n", encoding="utf-8"
    )

    result = compare_predictions_content(left, right)
    assert result["identical"] is False
    assert result["differing_fields"] == ["prediction"]
