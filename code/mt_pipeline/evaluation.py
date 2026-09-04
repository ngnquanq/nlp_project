from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import repo_path
from .io_utils import read_jsonl, sha256_file, write_json
from .normalize import score_form_chinese
from .schema import validate_prediction_rows


def _metrics():
    from sacrebleu.metrics import BLEU, CHRF

    return (
        BLEU(lowercase=False, tokenize="13a", smooth_method="exp", effective_order=False),
        CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False),
    )


def metrics_equivalent(
    stored: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    score_abs_tol: float = 1e-12,
) -> bool:
    """Compare metric payloads while allowing insignificant float-sum drift."""
    if not isinstance(stored, Mapping) or not isinstance(recomputed, Mapping):
        return False
    if stored.keys() != recomputed.keys():
        return False
    for metric_name, stored_metric in stored.items():
        recomputed_metric = recomputed[metric_name]
        if not isinstance(stored_metric, Mapping) or not isinstance(recomputed_metric, Mapping):
            return False
        if stored_metric.keys() != recomputed_metric.keys():
            return False
        for field, stored_value in stored_metric.items():
            recomputed_value = recomputed_metric[field]
            if field != "score":
                if stored_value != recomputed_value:
                    return False
                continue
            if (
                isinstance(stored_value, bool)
                or isinstance(recomputed_value, bool)
                or not isinstance(stored_value, (int, float))
                or not isinstance(recomputed_value, (int, float))
            ):
                return False
            stored_score = float(stored_value)
            recomputed_score = float(recomputed_value)
            if not math.isfinite(stored_score) or not math.isfinite(recomputed_score):
                return False
            if not math.isclose(
                stored_score,
                recomputed_score,
                rel_tol=0.0,
                abs_tol=score_abs_tol,
            ):
                return False
    return True


def _validated_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    splits = {row.get("split") for row in rows}
    if len(splits) != 1:
        raise ValueError("A prediction file must contain exactly one split")
    split = next(iter(splits))
    validate_prediction_rows(rows, expected_count=510 if split in {"val", "test"} else None)
    experiments = {row["experiment_id"] for row in rows}
    if len(experiments) != 1:
        raise ValueError("A prediction file must contain exactly one experiment")
    for row in rows:
        expected = score_form_chinese(row["prediction"])
        if row["prediction_scored"] != expected:
            raise ValueError(f"Non-canonical prediction_scored for {row['sample_id']}")
    return rows


def evaluate_predictions(prediction_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    path = repo_path(prediction_path)
    rows = _validated_rows(path)
    hypotheses = [row["prediction_scored"] for row in rows]
    references = [[score_form_chinese(row["reference"]) for row in rows]]
    bleu, chrf = _metrics()
    bleu_score = bleu.corpus_score(hypotheses, references)
    chrf_score = chrf.corpus_score(hypotheses, references)
    result = {
        "experiment_id": rows[0]["experiment_id"],
        "split": rows[0]["split"],
        "prediction_file": str(path),
        "prediction_sha256": sha256_file(path),
        "samples": len(rows),
        "metrics": {
            "sacrebleu": {
                "score": bleu_score.score,
                "signature": str(bleu.get_signature()),
                "verbose": str(bleu_score),
            },
            "chrf_pp": {
                "score": chrf_score.score,
                "signature": str(chrf.get_signature()),
                "verbose": str(chrf_score),
            },
        },
    }
    write_json(repo_path(output_path), result)
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def compare_predictions(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    samples: int = 1000,
    seed: int = 12345,
) -> dict[str, Any]:
    baseline_file, candidate_file = repo_path(baseline_path), repo_path(candidate_path)
    baseline, candidate = _validated_rows(baseline_file), _validated_rows(candidate_file)
    baseline_by_id = {row["sample_id"]: row for row in baseline}
    candidate_by_id = {row["sample_id"]: row for row in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Systems must contain the same sample IDs")
    ids = [row["sample_id"] for row in baseline]
    for sample_id in ids:
        if baseline_by_id[sample_id]["reference"] != candidate_by_id[sample_id]["reference"]:
            raise ValueError(f"Reference mismatch for {sample_id}")
    bleu, chrf = _metrics()
    metrics = {"sacrebleu": bleu, "chrf_pp": chrf}
    rng = random.Random(seed)
    results: dict[str, Any] = {}
    for name, metric in metrics.items():
        reference = [score_form_chinese(baseline_by_id[sample_id]["reference"]) for sample_id in ids]
        base_hyp = [baseline_by_id[sample_id]["prediction_scored"] for sample_id in ids]
        cand_hyp = [candidate_by_id[sample_id]["prediction_scored"] for sample_id in ids]
        observed_base = metric.corpus_score(base_hyp, [reference]).score
        observed_cand = metric.corpus_score(cand_hyp, [reference]).score
        deltas: list[float] = []
        for _ in range(samples):
            indices = [rng.randrange(len(ids)) for _ in ids]
            refs = [reference[index] for index in indices]
            base = [base_hyp[index] for index in indices]
            cand = [cand_hyp[index] for index in indices]
            deltas.append(
                metric.corpus_score(cand, [refs]).score
                - metric.corpus_score(base, [refs]).score
            )
        less_equal_zero = (sum(delta <= 0 for delta in deltas) + 1) / (samples + 1)
        greater_equal_zero = (sum(delta >= 0 for delta in deltas) + 1) / (samples + 1)
        results[name] = {
            "baseline_score": observed_base,
            "candidate_score": observed_cand,
            "observed_delta": observed_cand - observed_base,
            "delta_95_percent_interval": [
                _percentile(deltas, 0.025),
                _percentile(deltas, 0.975),
            ],
            "two_sided_p_value": min(1.0, 2 * min(less_equal_zero, greater_equal_zero)),
            "signature": str(metric.get_signature()),
        }
    output = {
        "baseline_experiment_id": baseline[0]["experiment_id"],
        "candidate_experiment_id": candidate[0]["experiment_id"],
        "split": baseline[0]["split"],
        "bootstrap_samples": samples,
        "seed": seed,
        "metrics": results,
    }
    write_json(repo_path(output_path), output)
    return output

