from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_yaml, repo_path
from .io_utils import write_json


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def project_status(config_paths: list[str | Path], output_path: str | Path | None = None) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    for config_path in config_paths:
        config = load_yaml(config_path)
        experiment_id = config["experiment_id"]
        work = repo_path(config.get("work_dir", f"work/{experiment_id}"))
        checkpoint = repo_path(config.get("checkpoint_dir", f"checkpoint/{experiment_id}"))
        predictions = repo_path(config.get("prediction_dir", "predictions"))
        experiments[experiment_id] = {
            "backend": config["backend"],
            "declared_status": config.get("status", "ready"),
            "run_manifest": _artifact(work / "run_manifest.json"),
            "checkpoint_directory": _artifact(checkpoint),
            "validation_predictions": _artifact(predictions / f"{experiment_id}.val.jsonl"),
            "selection_frozen": _artifact(work / "selection_frozen.json"),
            "test_predictions": _artifact(predictions / f"{experiment_id}.test.jsonl"),
            "validation_metrics": _artifact(repo_path("metrics") / f"{experiment_id}.val.json"),
            "test_metrics": _artifact(repo_path("metrics") / f"{experiment_id}.test.json"),
        }
    result = {
        "data_audit": _artifact(repo_path("metrics/data_audit.json")),
        "experiments": experiments,
        "error_analysis_summary": _artifact(repo_path("error_analysis/summary.json")),
        "report_pdf": _artifact(repo_path("report.pdf")),
        "slides_pdf": _artifact(repo_path("slides.pdf")),
    }
    if output_path is not None:
        write_json(repo_path(output_path), result)
    return result

