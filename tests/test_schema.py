import pytest

from mt_pipeline.config import REPO_ROOT, portable_project_path
from mt_pipeline.schema import PredictionRecord, validate_prediction_rows


def make_row(index: int = 1):
    return PredictionRecord(
        sample_id=f"test-{index:06d}",
        split="test",
        source="tôn mẹ",
        reference="尊母",
        prediction="尊母",
        prediction_raw="尊 母",
        prediction_scored="尊 母",
        experiment_id="example",
        model_name="model",
        checkpoint="checkpoint",
        decoding_config={"beam": 4},
    ).to_dict()


def test_prediction_schema_accepts_valid_row():
    validate_prediction_rows([make_row()])


def test_prediction_schema_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_rows([make_row(), make_row()])


def test_prediction_schema_rejects_empty_prediction():
    row = make_row()
    row["prediction"] = ""
    with pytest.raises(ValueError, match="prediction is empty"):
        PredictionRecord.from_dict(row)


def test_portable_project_path_removes_repository_prefix():
    checkpoint = REPO_ROOT / "checkpoint" / "example" / "checkpoint_best.pt"
    assert portable_project_path(checkpoint) == "checkpoint/example/checkpoint_best.pt"


def test_portable_project_path_masks_external_parent(tmp_path):
    checkpoint = tmp_path / "private" / "checkpoint_best.pt"
    assert portable_project_path(checkpoint) == "checkpoint_best.pt"

