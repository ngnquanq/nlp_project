import json

import pytest
import yaml

from mt_pipeline.freeze import ensure_selection_frozen, freeze_selection
from mt_pipeline.io_utils import write_json, write_jsonl
from mt_pipeline.schema import PredictionRecord


def test_freeze_locks_config_and_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_best.pt").write_bytes(b"checkpoint")
    config = {
        "experiment_id": "frozen-test",
        "backend": "fairseq",
        "checkpoint_dir": str(checkpoint_dir),
        "work_dir": str(tmp_path / "work"),
        "decoding": {"beam": 7},
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    rows = [
        PredictionRecord(
            sample_id=f"val-{index:06d}",
            split="val",
            source="nguồn",
            reference="漢文",
            prediction="漢文",
            prediction_raw="漢 文",
            prediction_scored="漢 文",
            experiment_id="frozen-test",
            model_name="model",
            checkpoint=str(checkpoint_dir / "checkpoint_best.pt"),
            decoding_config={"beam": 7},
        ).to_dict()
        for index in range(1, 511)
    ]
    predictions = tmp_path / "val.jsonl"
    metrics = tmp_path / "metrics.json"
    write_jsonl(predictions, rows)
    write_json(metrics, {"experiment_id": "frozen-test", "split": "val"})
    freeze_selection(config_path, predictions, metrics)
    assert ensure_selection_frozen(config_path)["test_generation_authorized"] is True

    config["decoding"]["beam"] = 5
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config changed"):
        ensure_selection_frozen(config_path)
