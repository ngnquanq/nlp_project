from mt_pipeline.error_analysis import prepare_annotation_template
from mt_pipeline.evaluation import compare_predictions
from mt_pipeline.io_utils import read_jsonl, write_jsonl
from mt_pipeline.schema import PredictionRecord


def _system_rows(experiment_id: str, prediction: str):
    return [
        PredictionRecord(
            sample_id=f"test-{index:06d}",
            split="test",
            source="nguồn",
            reference="漢文",
            prediction=prediction,
            prediction_raw=prediction,
            prediction_scored=" ".join(prediction),
            experiment_id=experiment_id,
            model_name="test-model",
            checkpoint="test-checkpoint",
            decoding_config={},
        ).to_dict()
        for index in range(1, 511)
    ]


def test_comparison_and_shared_error_sample(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    comparison = tmp_path / "comparison.json"
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(baseline, _system_rows("baseline", "漢文"))
    write_jsonl(candidate, _system_rows("candidate", "錯誤"))

    result = compare_predictions(baseline, candidate, comparison, samples=10, seed=12345)
    assert result["metrics"]["chrf_pp"]["observed_delta"] < 0

    prepare_annotation_template([baseline, candidate], annotations, sample_size=100, seed=42)
    rows = read_jsonl(annotations)
    assert len(rows) == 200
    assert len({row["sample_id"] for row in rows}) == 100
    assert {row["experiment_id"] for row in rows} == {"baseline", "candidate"}



def test_prepare_refuses_to_discard_existing_annotations(tmp_path):
    import pytest
    from mt_pipeline.error_analysis import _refuse_to_discard_annotations
    from mt_pipeline.io_utils import write_jsonl

    sheet = tmp_path / "manual_sample.jsonl"

    write_jsonl(sheet, [{"sample_id": "s-1", "annotation_status": "PENDING",
                         "annotator_1_labels": []}])
    _refuse_to_discard_annotations(sheet)  # untouched sheet may be rebuilt

    write_jsonl(sheet, [{"sample_id": "s-1", "annotation_status": "PENDING",
                         "annotator_1_labels": ["OMISSION"]}])
    with pytest.raises(RuntimeError, match="already carry manual annotations"):
        _refuse_to_discard_annotations(sheet)
