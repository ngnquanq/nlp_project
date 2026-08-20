"""Redirect an experiment into a parallel output tree and compare two runs.

Reruns must never overwrite the graded artifacts in ``checkpoint/``,
``predictions/`` and ``metrics/``. ``derive_run_config`` rewrites only the three
output-directory keys of a config, leaving every other key untouched, so a rerun
executes the same experiment into a disposable tree.

``compare_runs`` answers the question the reruns exist to answer: did two runs of
the same command produce the same model and the same translations? It compares
model weights and translation content, excluding state that is scoped to a
particular run (wall-clock timers, absolute paths, uninitialised device-tracking
buffers) and therefore differs even when the runs agree perfectly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import REPO_ROOT, load_yaml, repo_path
from .io_utils import write_json


OUTPUT_DIR_KEYS = ("work_dir", "checkpoint_dir", "prediction_dir")


def _relocate(value: str, root: Path) -> Path:
    """Rebase one output directory under ``root``, keeping its relative structure.

    The relative path must be preserved, not just the basename: configs name
    ``work/<id>`` and ``checkpoint/<id>``, which share a basename and would collapse
    into a single directory, putting the training log and the checkpoints in one place.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(REPO_ROOT)
        except ValueError:
            candidate = Path(candidate.name)
    return root / candidate


def derive_run_config(
    config_path: str | Path,
    run_root: str | Path,
    output_path: str | Path | None = None,
    gpu: bool = False,
) -> Path:
    """Copy a config with its output directories relocated beneath ``run_root``.

    ``gpu=True`` additionally forces ``training.cpu: false`` and
    ``training.fp16: true``. The smoke config runs on CPU in fp32, which exercises
    none of the CUDA determinism switches; the repro check needs the derived runs to
    take the same runtime path as E1/E3.
    """
    config = load_yaml(config_path)
    config.pop("_config_path", None)
    root = repo_path(run_root)
    for key in OUTPUT_DIR_KEYS:
        if key not in config:
            raise ValueError(f"Config is missing required output key: {key}")
        config[key] = str(_relocate(config[key], root))
    relocated = {config[key] for key in OUTPUT_DIR_KEYS}
    if len(relocated) != len(OUTPUT_DIR_KEYS):
        raise ValueError(f"Output directories collided after relocation: {sorted(relocated)}")
    if gpu:
        config.setdefault("training", {})
        config["training"]["cpu"] = False
        config["training"]["fp16"] = True
    destination = (
        repo_path(output_path)
        if output_path is not None
        else root / "configs" / Path(config_path).name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)
    return destination


# SinusoidalPositionalEmbedding allocates this as an uninitialized 1-element tensor
# purely to track device and dtype. It never enters the computation, and its garbage
# contents differ between processes, so comparing it says nothing about the model.
IGNORED_TENSOR_SUFFIXES = ("embed_positions._float_tensor",)

# Written by the predict step and scoped to the run that produced it: an absolute
# path that necessarily differs between two run trees. Every other field describes
# the translation itself and must match.
RUN_SCOPED_PREDICTION_FIELDS = ("checkpoint",)


def _model_state(checkpoint: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    if "model" not in payload:
        raise ValueError(f"Not a Fairseq checkpoint: {checkpoint}")
    return payload["model"]


def _is_ignored_tensor(name: str) -> bool:
    return name.endswith(IGNORED_TENSOR_SUFFIXES)


def compare_checkpoints(baseline: str | Path, candidate: str | Path) -> dict[str, Any]:
    """Compare two Fairseq checkpoints on model weights alone.

    Deliberately not a file hash. ``extra_state`` carries ``previous_training_time``
    (wall clock) and ``cfg``/``args`` embed the absolute save and data paths, so two
    perfectly deterministic runs produce different bytes. The weights are the claim.
    """
    import torch

    left = _model_state(repo_path(baseline))
    right = _model_state(repo_path(candidate))
    missing = sorted(set(left) ^ set(right))
    shared = sorted(name for name in set(left) & set(right) if not _is_ignored_tensor(name))
    ignored = sorted(name for name in set(left) & set(right) if _is_ignored_tensor(name))
    mismatched = sorted(name for name in shared if not torch.equal(left[name], right[name]))
    return {
        "tensors_compared": len(shared),
        "tensors_ignored": ignored,
        "keys_only_in_one": missing,
        "mismatched_tensors": mismatched,
        "identical": not missing and not mismatched,
    }


def compare_predictions_content(
    baseline: str | Path, candidate: str | Path
) -> dict[str, Any]:
    """Compare two prediction files on everything except run-scoped metadata."""
    from .io_utils import read_jsonl

    left = read_jsonl(repo_path(baseline))
    right = read_jsonl(repo_path(candidate))
    if len(left) != len(right):
        return {
            "rows": None,
            "identical": False,
            "error": f"row count differs: {len(left)} vs {len(right)}",
        }
    fields = sorted(set(left[0]) - set(RUN_SCOPED_PREDICTION_FIELDS)) if left else []
    differing = sorted(
        {field for a, b in zip(left, right) for field in fields if a[field] != b[field]}
    )
    return {
        "rows": len(left),
        "fields_compared": fields,
        "fields_ignored": list(RUN_SCOPED_PREDICTION_FIELDS),
        "differing_fields": differing,
        "identical": not differing,
    }


def compare_runs(
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    baseline_predictions: str | Path,
    candidate_predictions: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    weights = compare_checkpoints(baseline_checkpoint, candidate_checkpoint)
    predictions = compare_predictions_content(baseline_predictions, candidate_predictions)
    predictions["baseline"] = str(repo_path(baseline_predictions))
    predictions["candidate"] = str(repo_path(candidate_predictions))
    result = {
        "weights": weights,
        "predictions": predictions,
        "reproducible": weights["identical"] and predictions["identical"],
    }
    if output_path is not None:
        write_json(repo_path(output_path), result)
    return result
