from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .config import repo_path
from .io_utils import read_jsonl, write_jsonl
from .schema import validate_prediction_rows


ERROR_LABELS = {
    "CORRECT",
    "MISTRANSLATION",
    "OMISSION",
    "UNSUPPORTED_ADDITION",
    "ENTITY_ERROR",
    "NUMBER_OR_TIME_ERROR",
    "OTHER",
}


def prepare_annotation_template(
    prediction_paths: list[str | Path],
    output_path: str | Path,
    sample_size: int = 100,
    seed: int = 42,
) -> Path:
    if len(prediction_paths) < 2:
        raise ValueError("Error analysis requires at least two systems")
    systems: list[list[dict[str, Any]]] = []
    for value in prediction_paths:
        rows = read_jsonl(repo_path(value))
        validate_prediction_rows(rows, expected_count=510)
        systems.append(rows)
    ids = [row["sample_id"] for row in systems[0]]
    for rows in systems[1:]:
        if [row["sample_id"] for row in rows] != ids:
            raise ValueError("Prediction files must contain the same ordered sample IDs")
    if sample_size > len(ids):
        raise ValueError("Sample size exceeds prediction count")
    selected = set(random.Random(seed).sample(ids, sample_size))
    output_rows: list[dict[str, Any]] = []
    for rows in systems:
        for row in rows:
            if row["sample_id"] not in selected:
                continue
            output_rows.append({
                "sample_id": row["sample_id"],
                "experiment_id": row["experiment_id"],
                "source": row["source"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "annotation_status": "PENDING",
                "annotator_1_labels": [],
                "annotator_1_notes": "",
                "annotator_2_labels": [],
                "annotator_2_notes": "",
                "adjudicated_labels": [],
                "adjudication_notes": "",
            })
    output_rows.sort(key=lambda row: (int(row["sample_id"].split("-")[-1]), row["experiment_id"]))
    output = repo_path(output_path)
    _refuse_to_discard_annotations(output)
    write_jsonl(output, output_rows)
    return output


ANNOTATION_FIELDS = (
    "annotator_1_labels",
    "annotator_2_labels",
    "adjudicated_labels",
    "annotator_1_notes",
    "annotator_2_notes",
    "adjudication_notes",
)


def _refuse_to_discard_annotations(output: Path) -> None:
    """Regenerating the sheet resets every row to PENDING, discarding manual work."""
    if not output.exists():
        return
    try:
        existing = read_jsonl(output)
    except ValueError:
        return
    annotated = [
        row["sample_id"]
        for row in existing
        if any(row.get(field) for field in ANNOTATION_FIELDS)
        or row.get("annotation_status") not in (None, "PENDING")
    ]
    if annotated:
        raise RuntimeError(
            f"Refusing to overwrite {output}: {len(annotated)} row(s) already carry manual "
            f"annotations (first: {annotated[0]}) that regeneration would discard. "
            "Move the sheet aside first if you really mean to rebuild it."
        )


def validate_labels(labels: list[str]) -> None:
    unknown = set(labels) - ERROR_LABELS
    if unknown:
        raise ValueError(f"Unknown error labels: {sorted(unknown)}")
    if "CORRECT" in labels and len(labels) > 1:
        raise ValueError("CORRECT cannot coexist with error labels")


def summarize_annotations(annotation_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    from collections import Counter

    from .io_utils import write_json

    rows = read_jsonl(repo_path(annotation_path))
    if not rows:
        raise ValueError("Annotation file is empty")
    counts: dict[str, Counter[str]] = {}
    samples: dict[str, set[str]] = {}
    for row in rows:
        if row.get("annotation_status") != "ADJUDICATED":
            raise ValueError(f"Incomplete annotation: {row.get('sample_id')}")
        labels = row.get("adjudicated_labels", [])
        validate_labels(labels)
        experiment = row["experiment_id"]
        counts.setdefault(experiment, Counter()).update(labels)
        samples.setdefault(experiment, set()).add(row["sample_id"])
    if any(len(sample_ids) != 100 for sample_ids in samples.values()):
        raise ValueError("Each experiment must contain exactly 100 adjudicated sample IDs")
    sample_sets = list(samples.values())
    if any(sample_ids != sample_sets[0] for sample_ids in sample_sets[1:]):
        raise ValueError("Experiments must use the same 100 sample IDs")
    result = {
        "experiments": {
            experiment: {
                "samples": len(samples[experiment]),
                "label_counts": dict(counter),
                "label_percentages": {
                    label: 100 * count / len(samples[experiment])
                    for label, count in counter.items()
                },
            }
            for experiment, counter in counts.items()
        }
    }
    write_json(repo_path(output_path), result)
    return result
