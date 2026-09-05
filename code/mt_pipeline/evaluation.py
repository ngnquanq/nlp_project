from __future__ import annotations

import math
import random
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import repo_path
from .io_utils import read_jsonl, sha256_file, write_json
from .normalize import score_form_chinese
from .schema import validate_prediction_rows


DEFAULT_PROTOCOL = "moses-char-v1"
LEGACY_PROTOCOL = "13a-char-v1"
PROTOCOLS = (DEFAULT_PROTOCOL, LEGACY_PROTOCOL)


def _check_protocol(protocol: str) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown evaluation protocol: {protocol}")


@lru_cache(maxsize=1)
def _moses_tokenizer():
    from sacremoses import MosesTokenizer

    return MosesTokenizer(lang="zh")


def _bleu_inputs(texts: list[str], protocol: str) -> list[str]:
    _check_protocol(protocol)
    if protocol == LEGACY_PROTOCOL:
        return texts
    tokenizer = _moses_tokenizer()
    return [tokenizer.tokenize(text, return_str=True, escape=False,
                               aggressive_dash_splits=False) for text in texts]


def _bleu_metadata(protocol: str) -> dict[str, Any]:
    if protocol == LEGACY_PROTOCOL:
        return {}
    return {"preprocessing": {
        "protocol": DEFAULT_PROTOCOL,
        "implementation": "sacremoses.MosesTokenizer",
        "version": version("sacremoses"),
        "language": "zh",
        "input_form": "NFC; remove whitespace; space each Unicode code point",
        "escape": False,
        "aggressive_dash_splits": False,
    }}


def protocol_from_metrics(metrics: Mapping[str, Any]) -> str:
    """Identify stored scoring rules; never silently reinterpret an old bundle."""
    bleu = metrics["sacrebleu"]
    preprocessing = bleu.get("preprocessing")
    signature = bleu.get("signature", "").split("|")
    if preprocessing is None and "tok:13a" in signature:
        return LEGACY_PROTOCOL
    if (isinstance(preprocessing, Mapping)
            and preprocessing.get("protocol") == DEFAULT_PROTOCOL
            and "tok:none" in signature):
        return DEFAULT_PROTOCOL
    raise ValueError("Unrecognized stored BLEU protocol or missing Moses provenance")


def _metrics(protocol: str = DEFAULT_PROTOCOL):
    from sacrebleu.metrics import BLEU, CHRF

    _check_protocol(protocol)
    return (
        BLEU(lowercase=False, tokenize="13a" if protocol == LEGACY_PROTOCOL else "none",
             smooth_method="exp", effective_order=False),
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


def evaluate_predictions(
    prediction_path: str | Path, output_path: str | Path,
    protocol: str = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    path = repo_path(prediction_path)
    rows = _validated_rows(path)
    hypotheses = [row["prediction_scored"] for row in rows]
    references = [[score_form_chinese(row["reference"]) for row in rows]]
    bleu, chrf = _metrics(protocol)
    bleu_score = bleu.corpus_score(_bleu_inputs(hypotheses, protocol),
                                   [_bleu_inputs(references[0], protocol)])
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
                **_bleu_metadata(protocol),
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
    protocol: str = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    baseline_file, candidate_file = repo_path(baseline_path), repo_path(candidate_path)
    baseline, candidate = _validated_rows(baseline_file), _validated_rows(candidate_file)
    if baseline[0]["split"] != candidate[0]["split"]:
        raise ValueError("Systems must contain the same split")
    baseline_by_id = {row["sample_id"]: row for row in baseline}
    candidate_by_id = {row["sample_id"]: row for row in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Systems must contain the same sample IDs")
    ids = [row["sample_id"] for row in baseline]
    for sample_id in ids:
        if baseline_by_id[sample_id]["reference"] != candidate_by_id[sample_id]["reference"]:
            raise ValueError(f"Reference mismatch for {sample_id}")
    bleu, chrf = _metrics(protocol)
    metrics = {"sacrebleu": bleu, "chrf_pp": chrf}
    rng = random.Random(seed)
    results: dict[str, Any] = {}
    for name, metric in metrics.items():
        reference = [score_form_chinese(baseline_by_id[sample_id]["reference"]) for sample_id in ids]
        base_hyp = [baseline_by_id[sample_id]["prediction_scored"] for sample_id in ids]
        cand_hyp = [candidate_by_id[sample_id]["prediction_scored"] for sample_id in ids]
        if name == "sacrebleu":
            # Tokenize once before resampling. chrF++ keeps its historical inputs.
            reference = _bleu_inputs(reference, protocol)
            base_hyp = _bleu_inputs(base_hyp, protocol)
            cand_hyp = _bleu_inputs(cand_hyp, protocol)
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
            **(_bleu_metadata(protocol) if name == "sacrebleu" else {}),
        }
    output = {
        "baseline_experiment_id": baseline[0]["experiment_id"],
        "candidate_experiment_id": candidate[0]["experiment_id"],
        "split": baseline[0]["split"],
        "bootstrap_samples": samples,
        "seed": seed,
        "baseline_prediction_sha256": sha256_file(baseline_file),
        "candidate_prediction_sha256": sha256_file(candidate_file),
        "metrics": results,
    }
    write_json(repo_path(output_path), output)
    return output
