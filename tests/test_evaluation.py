import pytest

from mt_pipeline.evaluation import evaluate_predictions, metrics_equivalent
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
    assert "tok:none" in result["metrics"]["sacrebleu"]["signature"]
    assert result["metrics"]["sacrebleu"]["preprocessing"]["protocol"] == "moses-char-v1"
    assert "nw:2" in result["metrics"]["chrf_pp"]["signature"]


def test_metric_equivalence_allows_cross_python_float_drift():
    stored = {
        "sacrebleu": {
            "score": 29.6209774968959,
            "signature": "version:2.6.0",
            "verbose": "BLEU = 29.62",
        }
    }
    recomputed = {
        "sacrebleu": {
            "score": 29.620977496895886,
            "signature": "version:2.6.0",
            "verbose": "BLEU = 29.62",
        }
    }

    assert metrics_equivalent(stored, recomputed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", 29.621),
        ("signature", "version:2.5.1"),
        ("verbose", "BLEU = 29.63"),
    ],
)
def test_metric_equivalence_rejects_meaningful_changes(field, value):
    stored = {
        "sacrebleu": {
            "score": 29.6209774968959,
            "signature": "version:2.6.0",
            "verbose": "BLEU = 29.62",
        }
    }
    recomputed = {"sacrebleu": dict(stored["sacrebleu"])}
    recomputed["sacrebleu"][field] = value

    assert not metrics_equivalent(stored, recomputed)


def test_metric_equivalence_rejects_missing_fields():
    stored = {
        "sacrebleu": {
            "score": 29.6209774968959,
            "signature": "version:2.6.0",
            "verbose": "BLEU = 29.62",
        }
    }
    recomputed = {"sacrebleu": {"score": stored["sacrebleu"]["score"]}}

    assert not metrics_equivalent(stored, recomputed)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "29.62"])
def test_metric_equivalence_rejects_invalid_scores(score):
    stored = {
        "sacrebleu": {
            "score": 29.6209774968959,
            "signature": "version:2.6.0",
            "verbose": "BLEU = 29.62",
        }
    }
    recomputed = {"sacrebleu": {**stored["sacrebleu"], "score": score}}

    assert not metrics_equivalent(stored, recomputed)
