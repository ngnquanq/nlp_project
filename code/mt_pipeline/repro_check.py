"""Standing reproducibility gate over the artifacts already on disk.

Codifies the 2026-08-16 audit so its findings cannot silently regress:

* every frozen selection still matches its config and checkpoint;
* every stored metric file is still reproducible from its prediction file;
* every dataset split still hashes to what each run manifest recorded.

The E2 freeze break is asserted as a *known* state rather than ignored. See
``EXPECTED_FREEZE_STATE``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_yaml, repo_path
from .data import dataset_fingerprint
from .evaluation import evaluate_predictions, metrics_equivalent
from .freeze import ensure_selection_frozen


# E2's config was edited at 2026-08-16 10:03, twelve minutes after its test
# predictions were written, so its frozen chain no longer verifies and the config
# bytes behind BLEU 40.77 are unrecoverable. Asserting the failure keeps it visible.
# If E2 is legitimately retrained and re-frozen, flip this entry to True.
EXPECTED_FREEZE_STATE = {
    "configs/e1_fairseq.yaml": True,
    "configs/e2_qwen3_qlora.yaml": False,
    "configs/e3_custom_fairseq_knowledge.yaml": True,
    # Not yet run. `_frozen_record_exists` reports it as skipped until it is, then
    # this expectation applies.
    "configs/e4_qwen3_knowledge.yaml": True,
}

EXPERIMENTS = (
    "e1_fairseq_vi_zh_v1",
    "e2_qwen3_8b_qlora_vi_zh_v1",
    "e3_custom_fairseq_knowledge_vi_zh_v1",
    "e4_qwen3_knowledge_vi_zh_v1",
)


def _frozen_record_exists(config_path: str) -> bool:
    config = load_yaml(config_path)
    return (repo_path(config["work_dir"]) / "selection_frozen.json").exists()


def check_freeze_state() -> list[dict[str, Any]]:
    results = []
    for config_path, expected_pass in EXPECTED_FREEZE_STATE.items():
        # checkpoint/, predictions/ and work/ are generated artifacts that the repo
        # does not carry (README: submit them through the private channel). A fresh
        # clone has nothing to verify, which is "skipped", not "failed".
        if not _frozen_record_exists(config_path):
            results.append({
                "config": config_path,
                "status": "skipped",
                "ok": True,
                "reason": "no frozen selection on disk; run the experiment first",
            })
            continue
        error = None
        try:
            ensure_selection_frozen(config_path)
            actual_pass = True
        except Exception as exc:  # noqa: BLE001 - the message is the finding
            actual_pass = False
            error = str(exc)
        results.append({
            "config": config_path,
            "status": "checked",
            "expected_pass": expected_pass,
            "actual_pass": actual_pass,
            "ok": actual_pass == expected_pass,
            "error": error,
        })
    return results


def check_stored_metrics(scratch_dir: str | Path) -> list[dict[str, Any]]:
    scratch = repo_path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for experiment_id in EXPERIMENTS:
        for split in ("val", "test"):
            predictions = repo_path(f"predictions/{experiment_id}.{split}.jsonl")
            stored_path = repo_path(f"metrics/{experiment_id}.{split}.json")
            if not predictions.exists() or not stored_path.exists():
                results.append({
                    "experiment_id": experiment_id,
                    "split": split,
                    "status": "skipped",
                    "ok": True,
                    "reason": "no stored predictions/metrics on disk",
                })
                continue
            recomputed = evaluate_predictions(
                predictions, scratch / f"{experiment_id}.{split}.json"
            )
            with stored_path.open(encoding="utf-8") as handle:
                stored = json.load(handle)
            results.append({
                "experiment_id": experiment_id,
                "split": split,
                "status": "checked",
                "ok": metrics_equivalent(stored["metrics"], recomputed["metrics"]),
                "sacrebleu": recomputed["metrics"]["sacrebleu"]["score"],
                "chrf_pp": recomputed["metrics"]["chrf_pp"]["score"],
            })
    return results


def check_dataset_hashes() -> list[dict[str, Any]]:
    results = []
    for experiment_id in EXPERIMENTS:
        manifest_path = repo_path(f"work/{experiment_id}/run_manifest.json")
        if not manifest_path.exists():
            results.append({
                "experiment_id": experiment_id,
                "status": "skipped",
                "ok": True,
                "reason": "no run_manifest.json on disk",
            })
            continue
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        config = load_yaml(manifest["config"]["_config_path"])
        current = dataset_fingerprint(config["dataset_config"])
        recorded = manifest["dataset"]
        results.append({
            "experiment_id": experiment_id,
            "status": "checked",
            "ok": (
                current["splits"] == recorded["splits"]
                # A real input for E3, and recorded by every manifest.
                and current["knowledge_sha256"] == recorded["knowledge_sha256"]
            ),
        })
    return results


def run_checks(scratch_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    report = {
        "freeze_state": check_freeze_state(),
        "stored_metrics": check_stored_metrics(scratch_dir),
        "dataset_hashes": check_dataset_hashes(),
    }
    report["ok"] = all(
        entry["ok"] for section in report.values() for entry in section
    )
    if output_path is not None:
        from .io_utils import write_json

        write_json(repo_path(output_path), report)
    return report
