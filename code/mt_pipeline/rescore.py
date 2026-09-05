"""Evaluation-only migration; original predictions, metrics and freezes are inputs."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import repo_path
from .evaluation import (
    DEFAULT_PROTOCOL, compare_predictions, evaluate_predictions,
    metrics_equivalent, protocol_from_metrics,
)
from .io_utils import sha256_file, write_json


def rescore_moses(
    prediction_dir: str | Path, metrics_dir: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    predictions, original, output = map(repo_path, (prediction_dir, metrics_dir, output_dir))
    if output.resolve() in {original.resolve(), predictions.resolve()}:
        raise ValueError("Moses output must be separate from original metrics and predictions")
    files = sorted(predictions.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No saved predictions in {predictions}")
    started = time.perf_counter()
    report: dict[str, Any] = {"protocol": DEFAULT_PROTOCOL, "scores": [], "comparisons": []}
    by_experiment = {}
    with tempfile.TemporaryDirectory(prefix="mt-moses-verify-") as scratch:
        for prediction in files:
            name = prediction.with_suffix(".json").name
            old_path = original / name
            old = None
            if old_path.exists():
                old = json.loads(old_path.read_text(encoding="utf-8"))
                previous = evaluate_predictions(
                    prediction, Path(scratch) / name, protocol_from_metrics(old["metrics"]),
                )
                if (old.get("prediction_sha256") != sha256_file(prediction)
                        or not metrics_equivalent(old["metrics"], previous["metrics"])):
                    raise ValueError(f"Original metrics do not reproduce: {old_path}")
            current = evaluate_predictions(prediction, output / name)
            by_experiment[current["experiment_id"], current["split"]] = prediction
            report["scores"].append({
                "file": name, "samples": current["samples"],
                "prediction_sha256": current["prediction_sha256"],
                "previous_protocol": protocol_from_metrics(old["metrics"]) if old else None,
                "previous_metric_sha256": sha256_file(old_path) if old else None,
                "bleu": current["metrics"]["sacrebleu"]["score"],
                "bleu_delta": (current["metrics"]["sacrebleu"]["score"]
                               - old["metrics"]["sacrebleu"]["score"]) if old else None,
                "chrf_pp": current["metrics"]["chrf_pp"]["score"],
                "chrf_pp_delta": (current["metrics"]["chrf_pp"]["score"]
                                  - old["metrics"]["chrf_pp"]["score"]) if old else None,
            })
    for stored_path in sorted(original.glob("*.json")):
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        if "baseline_experiment_id" not in stored or "bootstrap_samples" not in stored:
            continue
        baseline = by_experiment[stored["baseline_experiment_id"], stored["split"]]
        candidate = by_experiment[stored["candidate_experiment_id"], stored["split"]]
        print(f"Rescoring comparison {stored_path.name}", flush=True)
        compared = compare_predictions(
            baseline, candidate, output / stored_path.name,
            samples=stored["bootstrap_samples"], seed=stored["seed"],
        )
        report["comparisons"].append({
            "file": stored_path.name,
            "previous_comparison_sha256": sha256_file(stored_path),
            "bootstrap_samples": compared["bootstrap_samples"], "seed": compared["seed"],
            "metrics": compared["metrics"],
        })
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "migration_summary.json", report)
    return report
