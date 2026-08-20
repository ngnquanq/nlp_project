import pytest

from mt_pipeline.evaluation import evaluate_predictions
from mt_pipeline.io_utils import write_jsonl
from mt_pipeline.schema import PredictionRecord


def test_identical_predictions_score_perfectly(tmp_path):
    rows = []
    for index in range(1, 511):
        rows.append(
            PredictionRecord(
                sample_id=f"test-{index:06d}",
                split="test",
                source="một câu",
                reference="尊母黎氏",
                prediction="尊母黎氏",
                prediction_raw="尊母黎氏",
                prediction_scored="尊 母 黎 氏",
                experiment_id="perfect",
                model_name="test",
                checkpoint="test-checkpoint",
                decoding_config={},
            ).to_dict()
        )
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    write_jsonl(predictions, rows)
    result = evaluate_predictions(predictions, output)
    assert result["metrics"]["sacrebleu"]["score"] == pytest.approx(100.0)
    assert result["metrics"]["chrf_pp"]["score"] == pytest.approx(100.0)
    assert "tok:13a" in result["metrics"]["sacrebleu"]["signature"]
    assert "nw:2" in result["metrics"]["chrf_pp"]["signature"]
