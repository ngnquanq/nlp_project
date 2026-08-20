from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_yaml, repo_path
from .io_utils import read_jsonl, sha256_file, sha256_tree, write_json
from .schema import validate_prediction_rows


def _checkpoint(config: dict[str, Any]) -> Path:
    root = repo_path(config["checkpoint_dir"])
    if config["backend"] in {"fairseq", "fairseq_knowledge"}:
        return root / "checkpoint_best.pt"
    if config["backend"] in {"qlora", "qlora_knowledge"}:
        return root / "adapter"
    raise ValueError(f"Unsupported freeze backend: {config['backend']}")


def _checkpoint_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return sha256_tree(path)
    raise FileNotFoundError(f"Missing checkpoint: {path}")


def freeze_selection(
    config_path: str | Path,
    validation_predictions: str | Path,
    validation_metrics: str | Path,
) -> Path:
    config = load_yaml(config_path)
    prediction_path = repo_path(validation_predictions)
    metric_path = repo_path(validation_metrics)
    rows = read_jsonl(prediction_path)
    validate_prediction_rows(rows, expected_count=510)
    if {row["split"] for row in rows} != {"val"}:
        raise ValueError("Selection can only be frozen from validation predictions")
    if {row["experiment_id"] for row in rows} != {config["experiment_id"]}:
        raise ValueError("Validation predictions do not match the experiment config")
    with metric_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("experiment_id") != config["experiment_id"] or metrics.get("split") != "val":
        raise ValueError("Validation metrics do not match the experiment config")
    checkpoint = _checkpoint(config)
    record = {
        "experiment_id": config["experiment_id"],
        "config_file": str(repo_path(config_path)),
        "config_sha256": sha256_file(repo_path(config_path)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_hash(checkpoint),
        "validation_predictions": str(prediction_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "validation_metrics": str(metric_path),
        "validation_metrics_sha256": sha256_file(metric_path),
        "decoding_config": config["decoding"],
        "test_generation_authorized": True,
    }
    output = repo_path(config["work_dir"]) / "selection_frozen.json"
    write_json(output, record)
    return output


def ensure_selection_frozen(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    path = repo_path(config["work_dir"]) / "selection_frozen.json"
    if not path.exists():
        raise RuntimeError(
            "Test generation is locked. Evaluate validation and run the freeze-selection command first."
        )
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("config_sha256") != sha256_file(repo_path(config_path)):
        raise RuntimeError("Experiment config changed after validation selection was frozen")
    checkpoint = _checkpoint(config)
    if record.get("checkpoint_sha256") != _checkpoint_hash(checkpoint):
        raise RuntimeError("Checkpoint changed after validation selection was frozen")
    if record.get("decoding_config") != config["decoding"]:
        raise RuntimeError("Decoding configuration changed after validation selection was frozen")
    return record
